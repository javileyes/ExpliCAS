# Engine Improvement Scorecard

- Generated: 2026-06-06T05:33:21.927408+00:00
- Git branch: main
- Git commit: `234771497c1a496849e4118ee9316b604b47cbb6`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=2
- By area: calculus / integration:1, calculus / runtime:1
- Recent 1: `calculus / runtime` - 2026-06-06 - Discovery observe-only: shifted sqrt trig quotient simplification dominates verification
- Recent 2: `calculus / integration` - 2026-06-06 - Discovery observe-only: cosh^-4 direct primitive presentation shifts cost into verifier

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=343

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=800.84ms avg_case_ms=8.01 simplify=221.98ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=683.33ms avg_case_ms=3.42 simplify=224.85ms avg_simplify_ms=1.12, product total=100 failed=0 elapsed=456.87ms avg_case_ms=4.57 simplify=127.87ms avg_simplify_ms=1.28, difference total=50 failed=0 elapsed=318.56ms avg_case_ms=6.37 simplify=99.85ms avg_simplify_ms=2.00
- Engine hotspots: sum simplify=224.85ms avg_simplify_ms=1.12 wall=683.33ms, shifted_quotient simplify=221.98ms avg_simplify_ms=2.22 wall=800.84ms, product simplify=127.87ms avg_simplify_ms=1.28 wall=456.87ms, difference simplify=99.85ms avg_simplify_ms=2.00 wall=318.56ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=800.84ms avg_case_ms=8.01 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=499.20ms avg_case_ms=4.99 avg_simplify_ms=1.57, product@0+100 failed=0 elapsed=456.87ms avg_case_ms=4.57 avg_simplify_ms=1.28, difference@0+50 failed=0 elapsed=318.56ms avg_case_ms=6.37 avg_simplify_ms=2.00, sum@700+100 failed=0 elapsed=184.13ms avg_case_ms=1.84 avg_simplify_ms=0.68
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.49ms median_wire=12.55ms median_wall=47.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.34ms median_wire=11.38ms median_wall=43.32ms, difference@0+50 #174 difference runs=3 median_simplify=11.31ms median_wire=11.36ms median_wall=43.21ms, product@0+100 #175 product runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=43.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.74ms median_wire=11.84ms median_wall=42.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.52s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.27s | passed=1 failed=0 |
