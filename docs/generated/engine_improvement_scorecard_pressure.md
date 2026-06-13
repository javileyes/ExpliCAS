# Engine Improvement Scorecard

- Generated: 2026-06-13T09:22:37.506595+00:00
- Git branch: main
- Git commit: `b849994f2ed2db9b7cb797920074335143603b93`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=778.71ms avg_case_ms=7.79 simplify=221.81ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=707.99ms avg_case_ms=3.54 simplify=239.77ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=474.93ms avg_case_ms=4.75 simplify=136.68ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=329.30ms avg_case_ms=6.59 simplify=104.18ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=239.77ms avg_simplify_ms=1.20 wall=707.99ms, shifted_quotient simplify=221.81ms avg_simplify_ms=2.22 wall=778.71ms, product simplify=136.68ms avg_simplify_ms=1.37 wall=474.93ms, difference simplify=104.18ms avg_simplify_ms=2.08 wall=329.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=778.71ms avg_case_ms=7.79 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=514.93ms avg_case_ms=5.15 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=474.93ms avg_case_ms=4.75 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=329.30ms avg_case_ms=6.59 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=193.06ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.03ms median_wall=49.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.73ms median_wall=44.39ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=43.82ms, product@0+100 #175 product runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.48ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.51ms median_wire=10.58ms median_wall=40.65ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
