# Engine Improvement Scorecard

- Generated: 2026-07-28T10:38:45.734030+00:00
- Git branch: main
- Git commit: `afc17ad2bf15ccbc60135b2fecc57ac21a7986c0`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=957.05ms avg_case_ms=9.57 simplify=267.78ms avg_simplify_ms=2.68, sum total=200 failed=0 elapsed=839.64ms avg_case_ms=4.20 simplify=273.26ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=595.42ms avg_case_ms=5.95 simplify=171.05ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=385.99ms avg_case_ms=7.72 simplify=117.54ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=273.26ms avg_simplify_ms=1.37 wall=839.64ms, shifted_quotient simplify=267.78ms avg_simplify_ms=2.68 wall=957.05ms, product simplify=171.05ms avg_simplify_ms=1.71 wall=595.42ms, difference simplify=117.54ms avg_simplify_ms=2.35 wall=385.99ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=957.05ms avg_case_ms=9.57 avg_simplify_ms=2.68, sum@0+100 failed=0 elapsed=613.37ms avg_case_ms=6.13 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=595.42ms avg_case_ms=5.95 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=385.99ms avg_case_ms=7.72 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=226.27ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.44ms median_wall=62.82ms, product@0+100 #175 product runs=3 median_simplify=14.77ms median_wire=14.81ms median_wall=56.91ms, sum@0+100 #173 sum runs=3 median_simplify=15.03ms median_wire=15.08ms median_wall=56.99ms, difference@0+50 #174 difference runs=3 median_simplify=15.20ms median_wire=15.25ms median_wall=58.06ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.00ms median_wire=13.07ms median_wall=49.09ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.28s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
