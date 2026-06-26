# Engine Improvement Scorecard

- Generated: 2026-06-26T07:54:29.730914+00:00
- Git branch: main
- Git commit: `00aa4f9e432a60223fc8bbcdd52b64a5f765083f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.05ms avg_case_ms=7.86 simplify=224.35ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=714.79ms avg_case_ms=3.57 simplify=247.96ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=483.98ms avg_case_ms=4.84 simplify=142.30ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=344.28ms avg_case_ms=6.89 simplify=112.28ms avg_simplify_ms=2.25
- Engine hotspots: sum simplify=247.96ms avg_simplify_ms=1.24 wall=714.79ms, shifted_quotient simplify=224.35ms avg_simplify_ms=2.24 wall=786.05ms, product simplify=142.30ms avg_simplify_ms=1.42 wall=483.98ms, difference simplify=112.28ms avg_simplify_ms=2.25 wall=344.28ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.05ms avg_case_ms=7.86 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=513.23ms avg_case_ms=5.13 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=483.98ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=344.28ms avg_case_ms=6.89 avg_simplify_ms=2.25, sum@700+100 failed=0 elapsed=201.56ms avg_case_ms=2.02 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=49.09ms, difference@0+50 #174 difference runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.09ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.45ms, sum@0+100 #173 sum runs=3 median_simplify=11.81ms median_wire=11.87ms median_wall=45.14ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.98ms median_wire=11.06ms median_wall=40.66ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
