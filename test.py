import evaluate

metric = evaluate.load("human_ai_trust")

out = metric.compute(
    predictions=[1, 0, 1],
    references=[1, 1, 0],
    confidences=[0.9, 0.7, 0.8],
    human_trust_scores=[0.85, 0.6, 0.75],
)

print(out)
