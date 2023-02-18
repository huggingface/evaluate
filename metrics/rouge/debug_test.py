import evaluate

rouge = evaluate.load("rouge")
rouge.add(predictions="sentence 1", references="sentence 1")
print(rouge.compute())

# rouge = evaluate.load("rouge")
# print("Adding predictions and references")
# print(rouge.features)
# predictions = ["hello there", "general kenobi"]
# references = [["hello", "there"], ["general kenobi", "general yoda"]]
# for sample_pred, sample_ref in zip(predictions, references):
#     rouge.add(predictions=sample_pred, references=sample_ref)
# # rouge.add(predictions= 0, references= 0)
# print("Computing metrics")
# print(rouge.compute())

# print("Adding predictions and references")
# predictions = ["hello there", "general kenobi"]
# references = [["hello", "there"], ["general kenobi", "general yoda"]]
# results = rouge.compute(predictions=predictions,
#                        references=references)
# print(results)