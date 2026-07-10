# Engine Improvement Scorecard

- Generated: 2026-07-10T10:10:53.625404+00:00
- Git branch: main
- Git commit: `9df74a5d4d65173396f7e030ce384d652392575a`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=950.29ms avg_case_ms=9.50 simplify=263.21ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=823.34ms avg_case_ms=4.12 simplify=266.12ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=583.56ms avg_case_ms=5.84 simplify=165.50ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=398.32ms avg_case_ms=7.97 simplify=120.04ms avg_simplify_ms=2.40
- Engine hotspots: sum simplify=266.12ms avg_simplify_ms=1.33 wall=823.34ms, shifted_quotient simplify=263.21ms avg_simplify_ms=2.63 wall=950.29ms, product simplify=165.50ms avg_simplify_ms=1.66 wall=583.56ms, difference simplify=120.04ms avg_simplify_ms=2.40 wall=398.32ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=950.29ms avg_case_ms=9.50 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=601.01ms avg_case_ms=6.01 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=583.56ms avg_case_ms=5.84 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=398.32ms avg_case_ms=7.97 avg_simplify_ms=2.40, sum@700+100 failed=0 elapsed=222.33ms avg_case_ms=2.22 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.60ms median_wire=16.66ms median_wall=62.63ms, difference@0+50 #174 difference runs=3 median_simplify=14.87ms median_wire=14.92ms median_wall=56.74ms, product@0+100 #175 product runs=3 median_simplify=14.59ms median_wire=14.64ms median_wall=55.91ms, sum@0+100 #173 sum runs=3 median_simplify=14.53ms median_wire=14.57ms median_wall=56.11ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.51ms median_wire=12.58ms median_wall=47.91ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
