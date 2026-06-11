# Engine Improvement Scorecard

- Generated: 2026-06-11T06:28:37.990188+00:00
- Git branch: main
- Git commit: `1d70c6a275710f3433200800e4f2933f84fd7d84`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=773.03ms avg_case_ms=7.73 simplify=218.80ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=696.61ms avg_case_ms=3.48 simplify=233.62ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=467.58ms avg_case_ms=4.68 simplify=134.13ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=318.97ms avg_case_ms=6.38 simplify=101.26ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=233.62ms avg_simplify_ms=1.17 wall=696.61ms, shifted_quotient simplify=218.80ms avg_simplify_ms=2.19 wall=773.03ms, product simplify=134.13ms avg_simplify_ms=1.34 wall=467.58ms, difference simplify=101.26ms avg_simplify_ms=2.03 wall=318.97ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=773.03ms avg_case_ms=7.73 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=508.64ms avg_case_ms=5.09 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=467.58ms avg_case_ms=4.68 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=318.97ms avg_case_ms=6.38 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=187.96ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.78ms median_wire=12.84ms median_wall=48.90ms, sum@0+100 #173 sum runs=3 median_simplify=11.45ms median_wire=11.50ms median_wall=43.90ms, product@0+100 #175 product runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.57ms, difference@0+50 #174 difference runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=44.02ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.46ms median_wire=11.55ms median_wall=44.24ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.16s | passed=1 failed=0 |
