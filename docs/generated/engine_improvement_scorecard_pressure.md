# Engine Improvement Scorecard

- Generated: 2026-07-23T17:31:16.973234+00:00
- Git branch: main
- Git commit: `ad5bb41b7425723bdba52f277ee4d88979a00b34`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=989.47ms avg_case_ms=9.89 simplify=278.42ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=897.98ms avg_case_ms=4.49 simplify=288.16ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=624.41ms avg_case_ms=6.24 simplify=181.58ms avg_simplify_ms=1.82, difference total=50 failed=0 elapsed=408.88ms avg_case_ms=8.18 simplify=125.80ms avg_simplify_ms=2.52
- Engine hotspots: sum simplify=288.16ms avg_simplify_ms=1.44 wall=897.98ms, shifted_quotient simplify=278.42ms avg_simplify_ms=2.78 wall=989.47ms, product simplify=181.58ms avg_simplify_ms=1.82 wall=624.41ms, difference simplify=125.80ms avg_simplify_ms=2.52 wall=408.88ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=989.47ms avg_case_ms=9.89 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=662.85ms avg_case_ms=6.63 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=624.41ms avg_case_ms=6.24 avg_simplify_ms=1.82, difference@0+50 failed=0 elapsed=408.88ms avg_case_ms=8.18 avg_simplify_ms=2.52, sum@700+100 failed=0 elapsed=235.14ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.00ms median_wire=17.07ms median_wall=65.00ms, difference@0+50 #174 difference runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=58.66ms, product@0+100 #175 product runs=3 median_simplify=15.26ms median_wire=15.31ms median_wall=59.54ms, sum@0+100 #173 sum runs=3 median_simplify=17.17ms median_wire=17.28ms median_wall=62.80ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.15ms median_wire=13.24ms median_wall=50.02ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.92s | passed=450 failed=0 total=450 avg_case=6.489ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.79s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.95s | passed=1 failed=0 |
