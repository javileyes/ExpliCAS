# Engine Improvement Scorecard

- Generated: 2026-06-11T23:50:44.313778+00:00
- Git branch: main
- Git commit: `c7f2f45e041bb4e6b0a7af02faec71d7fae80f95`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=772.81ms avg_case_ms=7.73 simplify=218.49ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=705.53ms avg_case_ms=3.53 simplify=238.26ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=469.80ms avg_case_ms=4.70 simplify=134.90ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=327.66ms avg_case_ms=6.55 simplify=104.37ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=238.26ms avg_simplify_ms=1.19 wall=705.53ms, shifted_quotient simplify=218.49ms avg_simplify_ms=2.18 wall=772.81ms, product simplify=134.90ms avg_simplify_ms=1.35 wall=469.80ms, difference simplify=104.37ms avg_simplify_ms=2.09 wall=327.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=772.81ms avg_case_ms=7.73 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=515.56ms avg_case_ms=5.16 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=469.80ms avg_case_ms=4.70 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=327.66ms avg_case_ms=6.55 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=189.97ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.84ms median_wire=12.91ms median_wall=48.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=44.07ms, difference@0+50 #174 difference runs=3 median_simplify=11.54ms median_wire=11.58ms median_wall=43.70ms, product@0+100 #175 product runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.23ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.51ms median_wire=10.58ms median_wall=39.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.18s | passed=1 failed=0 |
