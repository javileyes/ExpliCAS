# Engine Improvement Scorecard

- Generated: 2026-07-17T11:35:18.583199+00:00
- Git branch: main
- Git commit: `963ce86565321e07b8cb7c9930cd635b85af4192`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.11 simplify=283.13ms avg_simplify_ms=2.83, sum total=200 failed=0 elapsed=893.98ms avg_case_ms=4.47 simplify=288.74ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=624.85ms avg_case_ms=6.25 simplify=180.20ms avg_simplify_ms=1.80, difference total=50 failed=0 elapsed=408.19ms avg_case_ms=8.16 simplify=125.06ms avg_simplify_ms=2.50
- Engine hotspots: sum simplify=288.74ms avg_simplify_ms=1.44 wall=893.98ms, shifted_quotient simplify=283.13ms avg_simplify_ms=2.83 wall=1.01s, product simplify=180.20ms avg_simplify_ms=1.80 wall=624.85ms, difference simplify=125.06ms avg_simplify_ms=2.50 wall=408.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.11 avg_simplify_ms=2.83, sum@0+100 failed=0 elapsed=658.71ms avg_case_ms=6.59 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=624.85ms avg_case_ms=6.25 avg_simplify_ms=1.80, difference@0+50 failed=0 elapsed=408.19ms avg_case_ms=8.16 avg_simplify_ms=2.50, sum@700+100 failed=0 elapsed=235.27ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.04ms median_wire=17.12ms median_wall=65.46ms, sum@0+100 #173 sum runs=3 median_simplify=15.33ms median_wire=15.40ms median_wall=58.76ms, product@0+100 #175 product runs=3 median_simplify=15.77ms median_wire=15.82ms median_wall=60.87ms, difference@0+50 #174 difference runs=3 median_simplify=15.65ms median_wire=15.70ms median_wall=58.86ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.89ms median_wire=12.97ms median_wall=49.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.94s | passed=450 failed=0 total=450 avg_case=6.533ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.95s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
