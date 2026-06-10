# Engine Improvement Scorecard

- Generated: 2026-06-10T10:20:03.060860+00:00
- Git branch: main
- Git commit: `d1103bb7f8cac0ba54e41982fa6f06c8da1a39ac`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.35ms avg_case_ms=8.07 simplify=233.80ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=718.52ms avg_case_ms=3.59 simplify=237.85ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=481.45ms avg_case_ms=4.81 simplify=138.60ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=336.05ms avg_case_ms=6.72 simplify=106.70ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=237.85ms avg_simplify_ms=1.19 wall=718.52ms, shifted_quotient simplify=233.80ms avg_simplify_ms=2.34 wall=807.35ms, product simplify=138.60ms avg_simplify_ms=1.39 wall=481.45ms, difference simplify=106.70ms avg_simplify_ms=2.13 wall=336.05ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.35ms avg_case_ms=8.07 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=521.42ms avg_case_ms=5.21 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=481.45ms avg_case_ms=4.81 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=336.05ms avg_case_ms=6.72 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=197.10ms avg_case_ms=1.97 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.59ms median_wire=13.67ms median_wall=51.04ms, difference@0+50 #174 difference runs=3 median_simplify=11.92ms median_wire=11.98ms median_wall=45.49ms, product@0+100 #175 product runs=3 median_simplify=11.89ms median_wire=11.95ms median_wall=45.31ms, sum@0+100 #173 sum runs=3 median_simplify=12.14ms median_wire=12.20ms median_wall=45.92ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.61ms median_wire=11.69ms median_wall=42.22ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.87s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.57s | passed=1 failed=0 |
