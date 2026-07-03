# Engine Improvement Scorecard

- Generated: 2026-07-03T23:47:33.342861+00:00
- Git branch: main
- Git commit: `eba3e8158f66ad2e29f8b2abfff4f07cdca854d7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=932.06ms avg_case_ms=9.32 simplify=259.52ms avg_simplify_ms=2.60, sum total=200 failed=0 elapsed=826.71ms avg_case_ms=4.13 simplify=267.25ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=572.65ms avg_case_ms=5.73 simplify=163.03ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=387.17ms avg_case_ms=7.74 simplify=117.44ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=267.25ms avg_simplify_ms=1.34 wall=826.71ms, shifted_quotient simplify=259.52ms avg_simplify_ms=2.60 wall=932.06ms, product simplify=163.03ms avg_simplify_ms=1.63 wall=572.65ms, difference simplify=117.44ms avg_simplify_ms=2.35 wall=387.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=932.06ms avg_case_ms=9.32 avg_simplify_ms=2.60, sum@0+100 failed=0 elapsed=602.69ms avg_case_ms=6.03 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=572.65ms avg_case_ms=5.73 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=387.17ms avg_case_ms=7.74 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=224.03ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.24ms median_wire=16.30ms median_wall=62.83ms, sum@0+100 #173 sum runs=3 median_simplify=14.80ms median_wire=14.85ms median_wall=56.35ms, product@0+100 #175 product runs=3 median_simplify=14.55ms median_wire=14.60ms median_wall=56.16ms, difference@0+50 #174 difference runs=3 median_simplify=14.61ms median_wire=14.66ms median_wall=56.18ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.18ms median_wire=12.25ms median_wall=46.67ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
