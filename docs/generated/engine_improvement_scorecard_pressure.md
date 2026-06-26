# Engine Improvement Scorecard

- Generated: 2026-06-26T14:27:03.854407+00:00
- Git branch: main
- Git commit: `18b69d1bbf6e7280919f604b9943e6a908bd560d`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.98ms avg_case_ms=8.05 simplify=231.07ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=746.99ms avg_case_ms=3.73 simplify=258.76ms avg_simplify_ms=1.29, product total=100 failed=0 elapsed=499.38ms avg_case_ms=4.99 simplify=147.90ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=345.48ms avg_case_ms=6.91 simplify=112.57ms avg_simplify_ms=2.25
- Engine hotspots: sum simplify=258.76ms avg_simplify_ms=1.29 wall=746.99ms, shifted_quotient simplify=231.07ms avg_simplify_ms=2.31 wall=804.98ms, product simplify=147.90ms avg_simplify_ms=1.48 wall=499.38ms, difference simplify=112.57ms avg_simplify_ms=2.25 wall=345.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.98ms avg_case_ms=8.05 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=529.47ms avg_case_ms=5.29 avg_simplify_ms=1.75, product@0+100 failed=0 elapsed=499.38ms avg_case_ms=4.99 avg_simplify_ms=1.48, difference@0+50 failed=0 elapsed=345.48ms avg_case_ms=6.91 avg_simplify_ms=2.25, sum@700+100 failed=0 elapsed=217.52ms avg_case_ms=2.18 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.59ms median_wire=13.67ms median_wall=51.27ms, difference@0+50 #174 difference runs=3 median_simplify=12.15ms median_wire=12.20ms median_wall=46.28ms, product@0+100 #175 product runs=3 median_simplify=12.04ms median_wire=12.09ms median_wall=46.08ms, sum@0+100 #173 sum runs=3 median_simplify=12.11ms median_wire=12.16ms median_wall=45.93ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.95ms median_wire=11.02ms median_wall=41.30ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.40s | passed=450 failed=0 total=450 avg_case=5.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.11s | passed=1 failed=0 |
