# Engine Improvement Scorecard

- Generated: 2026-06-10T15:47:46.384375+00:00
- Git branch: main
- Git commit: `95f1343feb477c516f8150e8e144b90291d801fc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.24ms avg_case_ms=7.82 simplify=221.77ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=713.66ms avg_case_ms=3.57 simplify=237.29ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=476.59ms avg_case_ms=4.77 simplify=136.78ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=327.06ms avg_case_ms=6.54 simplify=104.10ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=237.29ms avg_simplify_ms=1.19 wall=713.66ms, shifted_quotient simplify=221.77ms avg_simplify_ms=2.22 wall=782.24ms, product simplify=136.78ms avg_simplify_ms=1.37 wall=476.59ms, difference simplify=104.10ms avg_simplify_ms=2.08 wall=327.06ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.24ms avg_case_ms=7.82 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=525.51ms avg_case_ms=5.26 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=476.59ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=327.06ms avg_case_ms=6.54 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=188.15ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.30ms median_wall=52.49ms, sum@0+100 #173 sum runs=3 median_simplify=11.76ms median_wire=11.82ms median_wall=44.79ms, product@0+100 #175 product runs=3 median_simplify=11.52ms median_wire=11.58ms median_wall=44.14ms, difference@0+50 #174 difference runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.53ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.75ms median_wire=11.84ms median_wall=41.58ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.35s | passed=1 failed=0 |
