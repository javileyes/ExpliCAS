# Engine Improvement Scorecard

- Generated: 2026-07-17T07:52:05.519963+00:00
- Git branch: main
- Git commit: `a08271fadbde2a87da737b32aa447eaa1f1f217e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=973.35ms avg_case_ms=9.73 simplify=272.94ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=884.51ms avg_case_ms=4.42 simplify=294.55ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=604.58ms avg_case_ms=6.05 simplify=173.47ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=401.16ms avg_case_ms=8.02 simplify=121.84ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=294.55ms avg_simplify_ms=1.47 wall=884.51ms, shifted_quotient simplify=272.94ms avg_simplify_ms=2.73 wall=973.35ms, product simplify=173.47ms avg_simplify_ms=1.73 wall=604.58ms, difference simplify=121.84ms avg_simplify_ms=2.44 wall=401.16ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=973.35ms avg_case_ms=9.73 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=650.23ms avg_case_ms=6.50 avg_simplify_ms=2.11, product@0+100 failed=0 elapsed=604.58ms avg_case_ms=6.05 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=401.16ms avg_case_ms=8.02 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=234.28ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.83ms median_wire=16.90ms median_wall=70.34ms, sum@0+100 #173 sum runs=3 median_simplify=15.38ms median_wire=15.44ms median_wall=59.06ms, product@0+100 #175 product runs=3 median_simplify=15.38ms median_wire=15.44ms median_wall=58.63ms, difference@0+50 #174 difference runs=3 median_simplify=15.32ms median_wire=15.37ms median_wall=58.67ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.20ms median_wire=13.28ms median_wall=49.49ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
