# Engine Improvement Scorecard

- Generated: 2026-06-27T10:30:27.147790+00:00
- Git branch: main
- Git commit: `480126a8da348636e144a6f014551ac7c89cf031`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=797.44ms avg_case_ms=7.97 simplify=228.13ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=720.71ms avg_case_ms=3.60 simplify=253.98ms avg_simplify_ms=1.27, product total=100 failed=0 elapsed=485.94ms avg_case_ms=4.86 simplify=143.21ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=333.10ms avg_case_ms=6.66 simplify=108.21ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=253.98ms avg_simplify_ms=1.27 wall=720.71ms, shifted_quotient simplify=228.13ms avg_simplify_ms=2.28 wall=797.44ms, product simplify=143.21ms avg_simplify_ms=1.43 wall=485.94ms, difference simplify=108.21ms avg_simplify_ms=2.16 wall=333.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=797.44ms avg_case_ms=7.97 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=519.48ms avg_case_ms=5.19 avg_simplify_ms=1.74, product@0+100 failed=0 elapsed=485.94ms avg_case_ms=4.86 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=333.10ms avg_case_ms=6.66 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=201.24ms avg_case_ms=2.01 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.82ms, sum@0+100 #173 sum runs=3 median_simplify=11.55ms median_wire=11.61ms median_wall=44.11ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.94ms median_wall=44.95ms, product@0+100 #175 product runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=44.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.81ms median_wall=40.51ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
