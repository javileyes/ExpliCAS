# Engine Improvement Scorecard

- Generated: 2026-06-27T01:33:37.463975+00:00
- Git branch: main
- Git commit: `ccabf1089a80f4b1f2ea15c05cbc64efd4b6bcaa`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=781.15ms avg_case_ms=7.81 simplify=223.27ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=703.81ms avg_case_ms=3.52 simplify=242.51ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=481.88ms avg_case_ms=4.82 simplify=141.46ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=327.01ms avg_case_ms=6.54 simplify=105.74ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=242.51ms avg_simplify_ms=1.21 wall=703.81ms, shifted_quotient simplify=223.27ms avg_simplify_ms=2.23 wall=781.15ms, product simplify=141.46ms avg_simplify_ms=1.41 wall=481.88ms, difference simplify=105.74ms avg_simplify_ms=2.11 wall=327.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=781.15ms avg_case_ms=7.81 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=507.67ms avg_case_ms=5.08 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=481.88ms avg_case_ms=4.82 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=327.01ms avg_case_ms=6.54 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=196.14ms avg_case_ms=1.96 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.94ms median_wire=13.01ms median_wall=49.50ms, difference@0+50 #174 difference runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=44.58ms, sum@0+100 #173 sum runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.45ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=44.18ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.67ms median_wire=10.73ms median_wall=40.24ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
