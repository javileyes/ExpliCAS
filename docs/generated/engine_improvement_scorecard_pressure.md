# Engine Improvement Scorecard

- Generated: 2026-06-21T19:29:46.191247+00:00
- Git branch: main
- Git commit: `5b2e870534faadd86afd52368f7b64334b162709`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.61ms avg_case_ms=7.95 simplify=226.19ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=697.28ms avg_case_ms=3.49 simplify=236.45ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=479.98ms avg_case_ms=4.80 simplify=137.90ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=345.95ms avg_case_ms=6.92 simplify=110.93ms avg_simplify_ms=2.22
- Engine hotspots: sum simplify=236.45ms avg_simplify_ms=1.18 wall=697.28ms, shifted_quotient simplify=226.19ms avg_simplify_ms=2.26 wall=794.61ms, product simplify=137.90ms avg_simplify_ms=1.38 wall=479.98ms, difference simplify=110.93ms avg_simplify_ms=2.22 wall=345.95ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.61ms avg_case_ms=7.95 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=503.64ms avg_case_ms=5.04 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=479.98ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=345.95ms avg_case_ms=6.92 avg_simplify_ms=2.22, sum@700+100 failed=0 elapsed=193.64ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.03ms median_wall=49.40ms, difference@0+50 #174 difference runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=44.20ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.08ms, product@0+100 #175 product runs=3 median_simplify=11.55ms median_wire=11.60ms median_wall=43.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.60ms median_wire=10.67ms median_wall=40.00ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
