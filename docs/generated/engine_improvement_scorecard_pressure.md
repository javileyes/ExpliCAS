# Engine Improvement Scorecard

- Generated: 2026-07-18T10:47:58.539515+00:00
- Git branch: main
- Git commit: `790a19363c649ded3a4d918a0e229bc4bb3530c7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=992.69ms avg_case_ms=9.93 simplify=280.04ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=879.18ms avg_case_ms=4.40 simplify=287.08ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=605.30ms avg_case_ms=6.05 simplify=174.38ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=401.75ms avg_case_ms=8.04 simplify=122.38ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=287.08ms avg_simplify_ms=1.44 wall=879.18ms, shifted_quotient simplify=280.04ms avg_simplify_ms=2.80 wall=992.69ms, product simplify=174.38ms avg_simplify_ms=1.74 wall=605.30ms, difference simplify=122.38ms avg_simplify_ms=2.45 wall=401.75ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=992.69ms avg_case_ms=9.93 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=649.64ms avg_case_ms=6.50 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=605.30ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=401.75ms avg_case_ms=8.04 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=229.54ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.14ms median_wire=15.18ms median_wall=58.20ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.08ms median_wire=17.15ms median_wall=65.09ms, difference@0+50 #174 difference runs=3 median_simplify=15.30ms median_wire=15.35ms median_wall=58.50ms, product@0+100 #175 product runs=3 median_simplify=15.60ms median_wire=15.65ms median_wall=58.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.74ms median_wire=12.82ms median_wall=48.76ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
