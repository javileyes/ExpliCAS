# Engine Improvement Scorecard

- Generated: 2026-05-12T13:56:53.845176+00:00
- Git branch: main
- Git commit: `6107514182d1cc83e32e01727ae15fcdf4a14e7f`
- Profile: `fast`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Calculus Contract Signal

- Dimension: public calculus behavior, result simplification, domain conditions, and step noise.
- Interpretation: small executable calculus vertical slices; failures should be classified before broadening pre-calculus rules.
- `diff`: passed=164 failed=0 ignored=1 filtered_out=0

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_add_small` | `pass` | 4.00s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=435 (100.0%) timeouts=0 |
| `contextual_strict_fast` | `pass` | 31.36s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=64 (100.0%) timeouts=0 |
| `contextual_radical_fast` | `pass` | 0.17s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=15 (100.0%) timeouts=0 |
| `calculus_diff_contract` | `pass` | 11.76s | passed=164 failed=0 |
