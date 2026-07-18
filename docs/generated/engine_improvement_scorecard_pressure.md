# Engine Improvement Scorecard

- Generated: 2026-07-18T13:31:16.110876+00:00
- Git branch: main
- Git commit: `6b8dd2da217951a1398d4524ee8eeb0e44337540`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=988.55ms avg_case_ms=9.89 simplify=274.60ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=875.14ms avg_case_ms=4.38 simplify=281.30ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=604.63ms avg_case_ms=6.05 simplify=173.78ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=403.03ms avg_case_ms=8.06 simplify=121.72ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=281.30ms avg_simplify_ms=1.41 wall=875.14ms, shifted_quotient simplify=274.60ms avg_simplify_ms=2.75 wall=988.55ms, product simplify=173.78ms avg_simplify_ms=1.74 wall=604.63ms, difference simplify=121.72ms avg_simplify_ms=2.43 wall=403.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=988.55ms avg_case_ms=9.89 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=644.80ms avg_case_ms=6.45 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=604.63ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=403.03ms avg_case_ms=8.06 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=230.34ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.40ms median_wire=15.45ms median_wall=58.07ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.75ms median_wire=16.82ms median_wall=64.42ms, difference@0+50 #174 difference runs=3 median_simplify=15.30ms median_wire=15.35ms median_wall=58.85ms, product@0+100 #175 product runs=3 median_simplify=15.21ms median_wire=15.27ms median_wall=57.89ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=48.94ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
