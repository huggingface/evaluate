import evaluate
import pandas as pd

metric = evaluate.load("human_ai_trust")

df = pd.read_csv("data/human_ai_trust_demo.csv")

out = metric.compute(
    predictions=df["prediction"].tolist(),
    references=df["reference"].tolist(),
    confidences=df["confidence"].tolist(),
    human_trust_scores=df["human_trust"].tolist(),
    belief_priors=df["belief_prior"].tolist(),
    belief_posteriors=df["belief_posterior"].tolist(),
    explanation_complexity=df["explanation_length"].tolist(),
)

print(out)

