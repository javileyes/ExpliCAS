# Engine Improvement Scorecard

- Generated: 2026-07-10T20:01:09.587534+00:00
- Git branch: main
- Git commit: `a1def7304b20e15445b5bce75683771cbe2ac435`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=948.02ms avg_case_ms=9.48 simplify=266.41ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=833.46ms avg_case_ms=4.17 simplify=269.66ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=583.28ms avg_case_ms=5.83 simplify=166.09ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=387.85ms avg_case_ms=7.76 simplify=117.75ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=269.66ms avg_simplify_ms=1.35 wall=833.46ms, shifted_quotient simplify=266.41ms avg_simplify_ms=2.66 wall=948.02ms, product simplify=166.09ms avg_simplify_ms=1.66 wall=583.28ms, difference simplify=117.75ms avg_simplify_ms=2.36 wall=387.85ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=948.02ms avg_case_ms=9.48 avg_simplify_ms=2.66, sum@0+100 failed=0 elapsed=611.94ms avg_case_ms=6.12 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=583.28ms avg_case_ms=5.83 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=387.85ms avg_case_ms=7.76 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=221.52ms avg_case_ms=2.22 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.60ms median_wire=16.68ms median_wall=63.75ms, product@0+100 #175 product runs=3 median_simplify=14.59ms median_wire=14.64ms median_wall=56.47ms, difference@0+50 #174 difference runs=3 median_simplify=15.06ms median_wire=15.11ms median_wall=57.32ms, sum@0+100 #173 sum runs=3 median_simplify=15.05ms median_wire=15.10ms median_wall=57.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=48.30ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.75s | passed=450 failed=0 total=450 avg_case=6.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
