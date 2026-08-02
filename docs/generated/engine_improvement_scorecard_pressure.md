# Engine Improvement Scorecard

- Generated: 2026-08-02T14:02:58.343604+00:00
- Git branch: main
- Git commit: `adaa78a6d0b52090a473089250cf56b160f2be1a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=986.83ms avg_case_ms=9.87 simplify=279.33ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=855.72ms avg_case_ms=4.28 simplify=275.90ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=605.43ms avg_case_ms=6.05 simplify=174.84ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=391.18ms avg_case_ms=7.82 simplify=115.70ms avg_simplify_ms=2.31
- Engine hotspots: shifted_quotient simplify=279.33ms avg_simplify_ms=2.79 wall=986.83ms, sum simplify=275.90ms avg_simplify_ms=1.38 wall=855.72ms, product simplify=174.84ms avg_simplify_ms=1.75 wall=605.43ms, difference simplify=115.70ms avg_simplify_ms=2.31 wall=391.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=986.83ms avg_case_ms=9.87 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=621.43ms avg_case_ms=6.21 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=605.43ms avg_case_ms=6.05 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=391.18ms avg_case_ms=7.82 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=234.30ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.99ms median_wire=17.06ms median_wall=64.81ms, product@0+100 #175 product runs=3 median_simplify=15.03ms median_wire=15.09ms median_wall=57.91ms, sum@0+100 #173 sum runs=3 median_simplify=15.25ms median_wire=15.30ms median_wall=58.35ms, difference@0+50 #174 difference runs=3 median_simplify=15.43ms median_wire=15.48ms median_wall=59.25ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.85ms median_wire=12.93ms median_wall=48.88ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.28s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.76s | passed=1 failed=0 |
