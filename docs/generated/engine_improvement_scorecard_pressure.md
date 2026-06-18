# Engine Improvement Scorecard

- Generated: 2026-06-18T14:26:57.721802+00:00
- Git branch: main
- Git commit: `782bfad2315071a865490f4b7aee53d0814471b5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=806.17ms avg_case_ms=8.06 simplify=233.18ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=701.15ms avg_case_ms=3.51 simplify=237.73ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=484.34ms avg_case_ms=4.84 simplify=141.95ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=331.80ms avg_case_ms=6.64 simplify=105.98ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=237.73ms avg_simplify_ms=1.19 wall=701.15ms, shifted_quotient simplify=233.18ms avg_simplify_ms=2.33 wall=806.17ms, product simplify=141.95ms avg_simplify_ms=1.42 wall=484.34ms, difference simplify=105.98ms avg_simplify_ms=2.12 wall=331.80ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=806.17ms avg_case_ms=8.06 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=508.42ms avg_case_ms=5.08 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=484.34ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=331.80ms avg_case_ms=6.64 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=192.74ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.24ms median_wire=13.32ms median_wall=50.45ms, sum@0+100 #173 sum runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=45.03ms, difference@0+50 #174 difference runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.91ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.61ms, shifted_quotient@0+100 #160 shifted_quotient runs=3 median_simplify=9.21ms median_wire=9.28ms median_wall=34.96ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
