"""Demo: Behavioral Drift metric — interactive test."""
import gradio as gr
import evaluate

drift = evaluate.load("./behavioral_drift.py")

def check(predictions_text, references_text):
    preds = [p.strip() for p in predictions_text.split("|||") if p.strip()]
    refs = [r.strip() for r in references_text.split("|||") if r.strip()]
    if not preds or not refs:
        return "Error: need at least one prediction and reference (separate multiple with |||)"
    if len(preds) != len(refs):
        return f"Error: predictions ({len(preds)}) and references ({len(refs)}) must have same count"
    r = drift.compute(predictions=preds, references=refs)
    return f"""drift_score: {r["drift_score"]}
self_bleu: {r["self_bleu"]}
digit_density: {r["digit_density"]} (baseline: {r["digit_density_baseline"]})
repetition_ratio: {r["repetition_ratio"]} (baseline: {r["repetition_ratio_baseline"]})
diagnosis: {r["diagnosis"]}"""

demo = gr.Interface(
    fn=check,
    inputs=[
        gr.Textbox(label="FT outputs (separate with |||)", value="正常的中文回答|||199999999999999"),
        gr.Textbox(label="Base outputs (separate with |||)", value="base输出A|||base输出B"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Behavioral Drift — Fine-Tuning Quality Metric",
    description="Detects output collapse invisible to loss curves. 0=collapse, 1=healthy.",
)
demo.launch()
