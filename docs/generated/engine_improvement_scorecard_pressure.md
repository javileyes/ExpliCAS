# Engine Improvement Scorecard

- Generated: 2026-07-19T19:39:53.313276+00:00
- Git branch: main
- Git commit: `c0b9c45affffe5978f33a6b680d030b193958f1e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=994.55ms avg_case_ms=9.95 simplify=276.90ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=886.67ms avg_case_ms=4.43 simplify=287.03ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=608.53ms avg_case_ms=6.09 simplify=174.84ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=398.96ms avg_case_ms=7.98 simplify=121.68ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=287.03ms avg_simplify_ms=1.44 wall=886.67ms, shifted_quotient simplify=276.90ms avg_simplify_ms=2.77 wall=994.55ms, product simplify=174.84ms avg_simplify_ms=1.75 wall=608.53ms, difference simplify=121.68ms avg_simplify_ms=2.43 wall=398.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=994.55ms avg_case_ms=9.95 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=652.98ms avg_case_ms=6.53 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=608.53ms avg_case_ms=6.09 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=398.96ms avg_case_ms=7.98 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=233.69ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.18ms median_wire=15.22ms median_wall=57.70ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.84ms median_wire=16.91ms median_wall=65.23ms, difference@0+50 #174 difference runs=3 median_simplify=15.22ms median_wire=15.27ms median_wall=58.19ms, product@0+100 #175 product runs=3 median_simplify=16.26ms median_wire=16.32ms median_wall=59.35ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.98ms median_wire=13.05ms median_wall=49.02ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
