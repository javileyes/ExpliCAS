# Engine Improvement Scorecard

- Generated: 2026-07-23T22:15:19.307216+00:00
- Git branch: main
- Git commit: `72609fc7117c374bdd80eecc971b3f09d749ad91`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.03s avg_case_ms=10.29 simplify=292.30ms avg_simplify_ms=2.92, sum total=200 failed=0 elapsed=910.80ms avg_case_ms=4.55 simplify=299.10ms avg_simplify_ms=1.50, product total=100 failed=0 elapsed=624.30ms avg_case_ms=6.24 simplify=181.12ms avg_simplify_ms=1.81, difference total=50 failed=0 elapsed=408.29ms avg_case_ms=8.17 simplify=126.20ms avg_simplify_ms=2.52
- Engine hotspots: sum simplify=299.10ms avg_simplify_ms=1.50 wall=910.80ms, shifted_quotient simplify=292.30ms avg_simplify_ms=2.92 wall=1.03s, product simplify=181.12ms avg_simplify_ms=1.81 wall=624.30ms, difference simplify=126.20ms avg_simplify_ms=2.52 wall=408.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.03s avg_case_ms=10.29 avg_simplify_ms=2.92, sum@0+100 failed=0 elapsed=672.84ms avg_case_ms=6.73 avg_simplify_ms=2.13, product@0+100 failed=0 elapsed=624.30ms avg_case_ms=6.24 avg_simplify_ms=1.81, difference@0+50 failed=0 elapsed=408.29ms avg_case_ms=8.17 avg_simplify_ms=2.52, sum@700+100 failed=0 elapsed=237.96ms avg_case_ms=2.38 avg_simplify_ms=0.86
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.86ms median_wire=15.91ms median_wall=60.42ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.47ms median_wire=17.55ms median_wall=65.85ms, product@0+100 #175 product runs=3 median_simplify=15.54ms median_wire=15.59ms median_wall=58.99ms, difference@0+50 #174 difference runs=3 median_simplify=15.42ms median_wire=15.48ms median_wall=58.99ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.13ms median_wall=50.08ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.97s | passed=450 failed=0 total=450 avg_case=6.600ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.65s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
