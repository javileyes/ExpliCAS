# Engine Improvement Scorecard

- Generated: 2026-06-24T16:33:45.730431+00:00
- Git branch: main
- Git commit: `681dc8194d341ae0102583dcc9257c60c666c5cd`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.05ms avg_case_ms=7.94 simplify=226.25ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=705.31ms avg_case_ms=3.53 simplify=238.97ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=478.93ms avg_case_ms=4.79 simplify=139.47ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=333.58ms avg_case_ms=6.67 simplify=106.00ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=238.97ms avg_simplify_ms=1.19 wall=705.31ms, shifted_quotient simplify=226.25ms avg_simplify_ms=2.26 wall=794.05ms, product simplify=139.47ms avg_simplify_ms=1.39 wall=478.93ms, difference simplify=106.00ms avg_simplify_ms=2.12 wall=333.58ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.05ms avg_case_ms=7.94 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=510.31ms avg_case_ms=5.10 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=478.93ms avg_case_ms=4.79 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=333.58ms avg_case_ms=6.67 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.00ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.20ms median_wire=13.26ms median_wall=49.39ms, product@0+100 #175 product runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=45.17ms, sum@0+100 #173 sum runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.44ms, difference@0+50 #174 difference runs=3 median_simplify=11.68ms median_wire=11.74ms median_wall=43.82ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.62ms median_wire=10.70ms median_wall=40.31ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
