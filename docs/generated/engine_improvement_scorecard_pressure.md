# Engine Improvement Scorecard

- Generated: 2026-07-18T16:00:04.481334+00:00
- Git branch: main
- Git commit: `922fdaccd78b167c3717a13807048a1ed85d8bac`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=988.34ms avg_case_ms=9.88 simplify=273.65ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=888.52ms avg_case_ms=4.44 simplify=286.83ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=604.06ms avg_case_ms=6.04 simplify=173.30ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=400.23ms avg_case_ms=8.00 simplify=121.84ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=286.83ms avg_simplify_ms=1.43 wall=888.52ms, shifted_quotient simplify=273.65ms avg_simplify_ms=2.74 wall=988.34ms, product simplify=173.30ms avg_simplify_ms=1.73 wall=604.06ms, difference simplify=121.84ms avg_simplify_ms=2.44 wall=400.23ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=988.34ms avg_case_ms=9.88 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=655.95ms avg_case_ms=6.56 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=604.06ms avg_case_ms=6.04 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=400.23ms avg_case_ms=8.00 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=232.58ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.55ms median_wire=17.62ms median_wall=67.61ms, sum@0+100 #173 sum runs=3 median_simplify=15.81ms median_wire=15.86ms median_wall=60.02ms, difference@0+50 #174 difference runs=3 median_simplify=15.77ms median_wire=15.81ms median_wall=59.43ms, product@0+100 #175 product runs=3 median_simplify=15.17ms median_wire=15.23ms median_wall=57.28ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.17ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
