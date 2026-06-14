# Engine Improvement Scorecard

- Generated: 2026-06-14T07:49:18.827946+00:00
- Git branch: main
- Git commit: `50d963d4b0bcc1c3a018ea8f30505fd379fa409e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=812.17ms avg_case_ms=8.12 simplify=232.72ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=694.31ms avg_case_ms=3.47 simplify=232.86ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=480.86ms avg_case_ms=4.81 simplify=138.30ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=336.77ms avg_case_ms=6.74 simplify=106.07ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=232.86ms avg_simplify_ms=1.16 wall=694.31ms, shifted_quotient simplify=232.72ms avg_simplify_ms=2.33 wall=812.17ms, product simplify=138.30ms avg_simplify_ms=1.38 wall=480.86ms, difference simplify=106.07ms avg_simplify_ms=2.12 wall=336.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=812.17ms avg_case_ms=8.12 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=498.63ms avg_case_ms=4.99 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=480.86ms avg_case_ms=4.81 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=336.77ms avg_case_ms=6.74 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.68ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.83ms median_wire=12.89ms median_wall=48.91ms, difference@0+50 #174 difference runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=44.42ms, sum@0+100 #173 sum runs=3 median_simplify=11.57ms median_wire=11.61ms median_wall=44.05ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.65ms, shifted_quotient@0+100 #160 shifted_quotient runs=3 median_simplify=9.13ms median_wire=9.21ms median_wall=34.29ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
