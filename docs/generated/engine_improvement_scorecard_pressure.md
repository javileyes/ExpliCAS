# Engine Improvement Scorecard

- Generated: 2026-06-23T23:22:21.862635+00:00
- Git branch: main
- Git commit: `bab69bcb13844e02f5c3d9c6426cf452a18a0254`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.15ms avg_case_ms=7.88 simplify=225.23ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=697.28ms avg_case_ms=3.49 simplify=236.69ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=475.10ms avg_case_ms=4.75 simplify=137.18ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=329.27ms avg_case_ms=6.59 simplify=105.08ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=236.69ms avg_simplify_ms=1.18 wall=697.28ms, shifted_quotient simplify=225.23ms avg_simplify_ms=2.25 wall=788.15ms, product simplify=137.18ms avg_simplify_ms=1.37 wall=475.10ms, difference simplify=105.08ms avg_simplify_ms=2.10 wall=329.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.15ms avg_case_ms=7.88 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=503.65ms avg_case_ms=5.04 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=475.10ms avg_case_ms=4.75 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=329.27ms avg_case_ms=6.59 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=193.62ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.62ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.36ms, sum@0+100 #173 sum runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=44.17ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=45.05ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.68ms median_wire=10.75ms median_wall=40.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
