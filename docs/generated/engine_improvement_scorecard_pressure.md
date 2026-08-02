# Engine Improvement Scorecard

- Generated: 2026-08-02T09:09:43.714169+00:00
- Git branch: main
- Git commit: `f5c847bc6559801d1cb66bd2ff23af52b000349a`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=980.02ms avg_case_ms=9.80 simplify=275.69ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=853.94ms avg_case_ms=4.27 simplify=275.67ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=604.67ms avg_case_ms=6.05 simplify=174.00ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=390.41ms avg_case_ms=7.81 simplify=115.55ms avg_simplify_ms=2.31
- Engine hotspots: shifted_quotient simplify=275.69ms avg_simplify_ms=2.76 wall=980.02ms, sum simplify=275.67ms avg_simplify_ms=1.38 wall=853.94ms, product simplify=174.00ms avg_simplify_ms=1.74 wall=604.67ms, difference simplify=115.55ms avg_simplify_ms=2.31 wall=390.41ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=980.02ms avg_case_ms=9.80 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=622.78ms avg_case_ms=6.23 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=604.67ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=390.41ms avg_case_ms=7.81 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=231.16ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.70ms median_wire=16.76ms median_wall=64.22ms, difference@0+50 #174 difference runs=3 median_simplify=15.24ms median_wire=15.29ms median_wall=57.96ms, sum@0+100 #173 sum runs=3 median_simplify=15.33ms median_wire=15.39ms median_wall=58.20ms, product@0+100 #175 product runs=3 median_simplify=15.19ms median_wire=15.24ms median_wall=57.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.04ms median_wall=49.65ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.29s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
