# Engine Improvement Scorecard

- Generated: 2026-06-17T16:38:54.389802+00:00
- Git branch: main
- Git commit: `9ac99898cd5f7a7853f4eff292ae9669e64ab531`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=798.91ms avg_case_ms=7.99 simplify=228.66ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=725.07ms avg_case_ms=3.63 simplify=247.11ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=480.63ms avg_case_ms=4.81 simplify=138.58ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=338.10ms avg_case_ms=6.76 simplify=107.60ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=247.11ms avg_simplify_ms=1.24 wall=725.07ms, shifted_quotient simplify=228.66ms avg_simplify_ms=2.29 wall=798.91ms, product simplify=138.58ms avg_simplify_ms=1.39 wall=480.63ms, difference simplify=107.60ms avg_simplify_ms=2.15 wall=338.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=798.91ms avg_case_ms=7.99 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=524.57ms avg_case_ms=5.25 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=480.63ms avg_case_ms=4.81 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=338.10ms avg_case_ms=6.76 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=200.51ms avg_case_ms=2.01 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.39ms median_wire=13.47ms median_wall=50.56ms, sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.22ms, difference@0+50 #174 difference runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=44.64ms, product@0+100 #175 product runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.68ms median_wire=10.75ms median_wall=40.55ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
