# Engine Improvement Scorecard

- Generated: 2026-06-10T09:15:09.373590+00:00
- Git branch: main
- Git commit: `89d217a77948b49fb55b48f1e161c6bda641f588`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.08ms avg_case_ms=7.88 simplify=224.04ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=705.19ms avg_case_ms=3.53 simplify=233.75ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=482.18ms avg_case_ms=4.82 simplify=138.50ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=331.10ms avg_case_ms=6.62 simplify=106.34ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=233.75ms avg_simplify_ms=1.17 wall=705.19ms, shifted_quotient simplify=224.04ms avg_simplify_ms=2.24 wall=788.08ms, product simplify=138.50ms avg_simplify_ms=1.39 wall=482.18ms, difference simplify=106.34ms avg_simplify_ms=2.13 wall=331.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.08ms avg_case_ms=7.88 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=516.59ms avg_case_ms=5.17 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=482.18ms avg_case_ms=4.82 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=331.10ms avg_case_ms=6.62 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=188.60ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.15ms median_wall=48.65ms, sum@0+100 #173 sum runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.72ms, difference@0+50 #174 difference runs=3 median_simplify=11.59ms median_wire=11.63ms median_wall=44.01ms, product@0+100 #175 product runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.55ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.77ms median_wire=10.85ms median_wall=43.34ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.39s | passed=1 failed=0 |
