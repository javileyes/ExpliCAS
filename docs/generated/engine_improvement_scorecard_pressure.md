# Engine Improvement Scorecard

- Generated: 2026-06-19T21:36:47.158470+00:00
- Git branch: main
- Git commit: `1cf7cae63faa85a0834cc59e9b78d68ac756c7a2`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=810.39ms avg_case_ms=8.10 simplify=232.87ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=708.08ms avg_case_ms=3.54 simplify=240.87ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=480.98ms avg_case_ms=4.81 simplify=138.42ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=329.14ms avg_case_ms=6.58 simplify=105.10ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=240.87ms avg_simplify_ms=1.20 wall=708.08ms, shifted_quotient simplify=232.87ms avg_simplify_ms=2.33 wall=810.39ms, product simplify=138.42ms avg_simplify_ms=1.38 wall=480.98ms, difference simplify=105.10ms avg_simplify_ms=2.10 wall=329.14ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=810.39ms avg_case_ms=8.10 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=514.38ms avg_case_ms=5.14 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=480.98ms avg_case_ms=4.81 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=329.14ms avg_case_ms=6.58 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=193.69ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.14ms median_wall=49.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.54ms, difference@0+50 #174 difference runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.37ms, product@0+100 #175 product runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=45.02ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.85ms median_wall=40.65ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
