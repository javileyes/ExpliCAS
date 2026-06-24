# Engine Improvement Scorecard

- Generated: 2026-06-24T12:01:46.301039+00:00
- Git branch: main
- Git commit: `19cfede0640cdbe410eaa03d280a3e9cfb7578fb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.20ms avg_case_ms=7.95 simplify=227.79ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=708.35ms avg_case_ms=3.54 simplify=240.03ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=485.65ms avg_case_ms=4.86 simplify=139.93ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=333.22ms avg_case_ms=6.66 simplify=107.08ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=240.03ms avg_simplify_ms=1.20 wall=708.35ms, shifted_quotient simplify=227.79ms avg_simplify_ms=2.28 wall=795.20ms, product simplify=139.93ms avg_simplify_ms=1.40 wall=485.65ms, difference simplify=107.08ms avg_simplify_ms=2.14 wall=333.22ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.20ms avg_case_ms=7.95 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=514.47ms avg_case_ms=5.14 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=485.65ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=333.22ms avg_case_ms=6.66 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=193.88ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.18ms median_wire=13.26ms median_wall=49.89ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=44.87ms, difference@0+50 #174 difference runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=44.71ms, sum@0+100 #173 sum runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=44.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.86ms median_wall=40.79ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
