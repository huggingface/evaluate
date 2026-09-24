# Behavioral Drift

**A fine-tuning quality metric that catches what loss curves miss.**

## The gap

Everyone checks loss after fine-tuning. But loss can drop while every output collapses to `"199999999999..."`. Perplexity won't flag this. BLEU won't flag this. Only reading actual outputs catches it — and nobody reads every output.

**Standard metrics measure token-level quality. Behavioral Drift measures output integrity.**

## How it complements existing tools

| Tool | Measures | Blind spot |
|------|----------|------------|
| Perplexity | Token prediction | Output coherence |
| BLEU/ROUGE | n-gram overlap | Mode collapse, degeneration |
| Loss curve | Convergence | Behavioral degradation |
| **drift_score** | Output diversity + integrity | — |

Not a replacement. An addition. Use alongside perplexity — not instead of it.

## Three signals

- **self-BLEU**: pairwise similarity among outputs (high = mode collapse)
- **digit density**: fraction of numeric chars (high = garbage output)
- **repetition ratio**: unique/total token ratio (low = looping)

=> **drift_score** (0-1): 1.0 = healthy, 0.0 = collapsed.

## Usage

```python
import evaluate
drift = evaluate.load("./behavioral_drift.py")
r = drift.compute(
    predictions=ft_outputs,
    references=base_outputs,
)
print(r["drift_score"])  # 0.95 = healthy, 0.05 = collapse
```

## Origin

Extracted from 6 failed LoRA experiments (Qwen2.5-0.5B/1.5B). Loss curves looked fine. Outputs were broken. The checks that finally caught the collapse were formalized into this reusable metric — so others don't debug the same way.

## License

MIT
