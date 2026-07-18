# Engine Improvement Scorecard

- Generated: 2026-07-18T08:34:46.653940+00:00
- Git branch: main
- Git commit: `3973d9ee2fa3103cb3ac907d22d63ebcc7b9c387`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=990.34ms avg_case_ms=9.90 simplify=281.84ms avg_simplify_ms=2.82, sum total=200 failed=0 elapsed=879.89ms avg_case_ms=4.40 simplify=283.59ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=603.91ms avg_case_ms=6.04 simplify=173.23ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=402.08ms avg_case_ms=8.04 simplify=122.45ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=283.59ms avg_simplify_ms=1.42 wall=879.89ms, shifted_quotient simplify=281.84ms avg_simplify_ms=2.82 wall=990.34ms, product simplify=173.23ms avg_simplify_ms=1.73 wall=603.91ms, difference simplify=122.45ms avg_simplify_ms=2.45 wall=402.08ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=990.34ms avg_case_ms=9.90 avg_simplify_ms=2.82, sum@0+100 failed=0 elapsed=647.81ms avg_case_ms=6.48 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=603.91ms avg_case_ms=6.04 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=402.08ms avg_case_ms=8.04 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=232.08ms avg_case_ms=2.32 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.94ms median_wire=17.01ms median_wall=65.04ms, sum@0+100 #173 sum runs=3 median_simplify=15.24ms median_wire=15.29ms median_wall=57.83ms, difference@0+50 #174 difference runs=3 median_simplify=15.45ms median_wire=15.51ms median_wall=60.20ms, product@0+100 #175 product runs=3 median_simplify=15.48ms median_wire=15.54ms median_wall=59.16ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.30ms median_wire=13.38ms median_wall=50.57ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.59s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
