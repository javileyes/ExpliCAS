# Engine Improvement Scorecard

- Generated: 2026-07-19T22:58:54.385761+00:00
- Git branch: main
- Git commit: `40d9670548230d5c94d1d921b2f39ef67291fb27`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=984.29ms avg_case_ms=9.84 simplify=275.33ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=888.27ms avg_case_ms=4.44 simplify=286.65ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=606.01ms avg_case_ms=6.06 simplify=174.07ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=401.01ms avg_case_ms=8.02 simplify=121.87ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=286.65ms avg_simplify_ms=1.43 wall=888.27ms, shifted_quotient simplify=275.33ms avg_simplify_ms=2.75 wall=984.29ms, product simplify=174.07ms avg_simplify_ms=1.74 wall=606.01ms, difference simplify=121.87ms avg_simplify_ms=2.44 wall=401.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=984.29ms avg_case_ms=9.84 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=654.19ms avg_case_ms=6.54 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=606.01ms avg_case_ms=6.06 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=401.01ms avg_case_ms=8.02 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=234.08ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.11ms median_wire=15.16ms median_wall=58.01ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.84ms median_wire=16.91ms median_wall=64.33ms, difference@0+50 #174 difference runs=3 median_simplify=15.11ms median_wire=15.16ms median_wall=57.96ms, product@0+100 #175 product runs=3 median_simplify=16.30ms median_wire=16.35ms median_wall=58.50ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.12ms median_wall=49.30ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
