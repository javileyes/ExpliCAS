# Engine Improvement Scorecard

- Generated: 2026-08-02T11:33:05.850593+00:00
- Git branch: main
- Git commit: `646a1e03a4e1b437a10345fcd8801edbee50545c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=978.55ms avg_case_ms=9.79 simplify=277.29ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=853.04ms avg_case_ms=4.27 simplify=275.80ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=606.28ms avg_case_ms=6.06 simplify=173.86ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=395.53ms avg_case_ms=7.91 simplify=116.13ms avg_simplify_ms=2.32
- Engine hotspots: shifted_quotient simplify=277.29ms avg_simplify_ms=2.77 wall=978.55ms, sum simplify=275.80ms avg_simplify_ms=1.38 wall=853.04ms, product simplify=173.86ms avg_simplify_ms=1.74 wall=606.28ms, difference simplify=116.13ms avg_simplify_ms=2.32 wall=395.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=978.55ms avg_case_ms=9.79 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=620.97ms avg_case_ms=6.21 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=606.28ms avg_case_ms=6.06 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=395.53ms avg_case_ms=7.91 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=232.07ms avg_case_ms=2.32 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.17ms median_wire=17.25ms median_wall=64.52ms, product@0+100 #175 product runs=3 median_simplify=15.33ms median_wire=15.39ms median_wall=58.41ms, difference@0+50 #174 difference runs=3 median_simplify=15.40ms median_wire=15.45ms median_wall=59.03ms, sum@0+100 #173 sum runs=3 median_simplify=14.97ms median_wire=15.02ms median_wall=58.59ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.98ms median_wall=49.41ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.26s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
