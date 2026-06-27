# Engine Improvement Scorecard

- Generated: 2026-06-27T01:02:04.673513+00:00
- Git branch: main
- Git commit: `5e5e27a92bd8dfde5a79bf6914f8d28a6e453e39`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.70ms avg_case_ms=7.88 simplify=225.65ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=702.52ms avg_case_ms=3.51 simplify=243.28ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=484.52ms avg_case_ms=4.85 simplify=142.30ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=327.22ms avg_case_ms=6.54 simplify=105.69ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=243.28ms avg_simplify_ms=1.22 wall=702.52ms, shifted_quotient simplify=225.65ms avg_simplify_ms=2.26 wall=787.70ms, product simplify=142.30ms avg_simplify_ms=1.42 wall=484.52ms, difference simplify=105.69ms avg_simplify_ms=2.11 wall=327.22ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.70ms avg_case_ms=7.88 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=505.67ms avg_case_ms=5.06 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=484.52ms avg_case_ms=4.85 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=327.22ms avg_case_ms=6.54 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=196.85ms avg_case_ms=1.97 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.01ms median_wire=13.08ms median_wall=49.89ms, difference@0+50 #174 difference runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=45.09ms, sum@0+100 #173 sum runs=3 median_simplify=12.00ms median_wire=12.05ms median_wall=45.24ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.62ms median_wire=10.70ms median_wall=40.20ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
