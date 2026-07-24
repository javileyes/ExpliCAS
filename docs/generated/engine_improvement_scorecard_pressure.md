# Engine Improvement Scorecard

- Generated: 2026-07-24T20:22:47.149467+00:00
- Git branch: main
- Git commit: `38abc665306b52d4bc8c45268c29546cd0747460`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=974.12ms avg_case_ms=9.74 simplify=271.91ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=855.23ms avg_case_ms=4.28 simplify=277.33ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=604.90ms avg_case_ms=6.05 simplify=173.03ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=397.24ms avg_case_ms=7.94 simplify=121.62ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=277.33ms avg_simplify_ms=1.39 wall=855.23ms, shifted_quotient simplify=271.91ms avg_simplify_ms=2.72 wall=974.12ms, product simplify=173.03ms avg_simplify_ms=1.73 wall=604.90ms, difference simplify=121.62ms avg_simplify_ms=2.43 wall=397.24ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=974.12ms avg_case_ms=9.74 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=624.63ms avg_case_ms=6.25 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=604.90ms avg_case_ms=6.05 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=397.24ms avg_case_ms=7.94 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=230.60ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.99ms median_wire=17.06ms median_wall=64.97ms, difference@0+50 #174 difference runs=3 median_simplify=15.30ms median_wire=15.35ms median_wall=58.90ms, product@0+100 #175 product runs=3 median_simplify=15.11ms median_wire=15.16ms median_wall=57.54ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.00ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.16ms median_wall=49.49ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.61s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
